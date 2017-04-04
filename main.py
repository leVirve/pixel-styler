import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import datasets as dset
from models import Discriminator, Generator

cudnn.benchmark = True

input_nc = output_nc = 3
ngf = ndf = 64
img_sz = 256

num_epochs = 200
batch_size = 64
η = 0.0002
β1 = 0.5
λ = 100

num_workers = 4

tf = transforms.Compose([
    # transforms.Scale(256),
    transforms.ToTensor(),
])

print('===> Load datasets')
train_dataset = dset.ImageFolderDataset(root='./datasets/facades/train',
                                        transform=tf)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           shuffle=True)

print('===> Build model')
generator = Generator(input_nc, output_nc, ngf).cuda()
discriminator = Discriminator(input_nc, output_nc, ndf).cuda()

print(generator)
print(discriminator)

print('===> Build loss and optimizers')
criterion = nn.BCELoss().cuda()
criterion_l1 = nn.L1Loss().cuda()

g_optimizer = torch.optim.Adam(
    generator.parameters(), lr=η, betas=(β1, 0.999))
d_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=η, betas=(β1, 0.999))

real_a = Variable(torch.FloatTensor(batch_size, input_nc, img_sz, img_sz).cuda())
real_b = Variable(torch.FloatTensor(batch_size, output_nc, img_sz, img_sz).cuda())
label = Variable(torch.FloatTensor(batch_size).cuda())
fake_b = None
real_ab = None
fake_ab = None
real_label, fake_label = 1, 0


def create_example(img_a, img_b):

    real_b.data.resize_(img_a.size()).copy_(img_a)
    real_a.data.resize_(img_b.size()).copy_(img_b)

    global real_ab, fake_b, fake_ab
    real_ab = torch.cat((real_a, real_b), 1)
    fake_b = generator(real_a)
    fake_ab = torch.cat((real_a, fake_b.detach()), 1)


def update_discriminator():
    ''' discriminator
        Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
    '''
    d_optimizer.zero_grad()

    # Train with real
    output = discriminator(real_ab)
    label.data.resize_(output.size()).fill_(real_label)
    d_real_loss = criterion(output, label)
    d_real_loss.backward()
    d_x = output.data.mean()

    # Train with fake
    output = discriminator(fake_ab)  # is fakeb detach ?
    label.data.resize_(output.size()).fill_(fake_label)
    d_fake_loss = criterion(output, label)
    d_fake_loss.backward()
    d_g_z1 = output.data.mean()

    d_loss = (d_real_loss + d_fake_loss) / 2.0

    d_optimizer.step()
    return (d_x, d_g_z1), d_loss


def update_generator():
    ''' generator
        Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
    '''
    generator.zero_grad()

    output = discriminator(fake_ab)
    label.data.resize_(output.size()).fill_(real_label)
    g_loss = criterion(output, label) + λ * criterion_l1(fake_b, real_b)
    g_loss.backward()
    d_g_z2 = output.data.mean()

    g_optimizer.step()
    return (d_g_z2), g_loss


def train():
    print('===> Train model')
    for epoch in range(num_epochs):
        for i, (img_a, img_b) in enumerate(train_loader):

            create_example(img_a, img_b)

            d_f1, d_loss = update_discriminator()
            d_f2, g_loss = update_generator()

            if (i + 1) % 5 == 0:
                print('Epoch [{:3d}/{}]({:02d}/{:02d}) => '
                      'Loss_D: {:.4f} Loss_G: {:.4f}'
                      ' D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
                        epoch + 1, num_epochs, i + 1,
                        len(train_dataset) // batch_size,
                        d_loss.data[0], g_loss.data[0], *d_f1, *d_f2))

        torchvision.utils.save_image(
            fake_b.data, './output/fake_samples_epoch%d.png' % (epoch + 1))

    torch.save(generator.state_dict(), './generator.pkl')
    torch.save(discriminator.state_dict(), './discriminator.pkl')
    torch.save(generator, './generator.pth')


def main():
    train()

if __name__ == '__main__':
    main()