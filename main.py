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

# random seed
torch.cuda.manual_seed(123)

print('===> Load datasets')
train_dataset = dset.ImageFolderDataset(root='./datasets/facades/train',
                                        transform=tf)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           shuffle=True)

print('===> Build model')
generator = Generator(3, 3, 64).cuda()
discriminator = Discriminator(3, 3, 64).cuda()

print(generator)
print(discriminator)

print('===> Build loss and optimizers')
criterion = nn.BCELoss().cuda()
criterion_l1 = nn.L1Loss().cuda()

g_optimizer = torch.optim.Adam(
    generator.parameters(), lr=η, betas=(β1, 0.999))
d_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=η, betas=(β1, 0.999))

print('===> Train model')

real_a = Variable(torch.FloatTensor(batch_size, input_nc, 256, 256).cuda())
real_b = Variable(torch.FloatTensor(batch_size, output_nc, 256, 256).cuda())
label = Variable(torch.FloatTensor(batch_size).cuda())
real_label, fake_label = 1, 0

for epoch in range(num_epochs):
    for i, (img_a, img_b) in enumerate(train_loader):

        ''' discriminator '''
        d_optimizer.zero_grad()
        real_a.data.resize_(img_b.size()).copy_(img_b)
        real_b.data.resize_(img_a.size()).copy_(img_a)

        # Train with real
        output = discriminator(torch.cat((real_a, real_b), 1))
        label.data.resize_(output.size()).fill_(real_label)
        d_real_loss = criterion(output, label)
        d_real_loss.backward()
        d_x_y = output.data.mean()

        # Train with fake
        fake_b = generator(real_a)
        output = discriminator(torch.cat((real_a, fake_b.detach()), 1))
        label.data.resize_(output.size()).fill_(fake_label)
        d_fake_loss = criterion(output, label)
        d_fake_loss.backward()
        d_x_gx = output.data.mean()

        d_loss = (d_real_loss + d_fake_loss) / 2.0

        d_optimizer.step()

        ''' generator '''
        generator.zero_grad()

        output = discriminator(torch.cat((real_a, fake_b), 1))
        label.data.resize_(output.size()).fill_(real_label)
        g_loss = criterion(output, label) + λ * criterion_l1(fake_b, real_b)
        g_loss.backward()
        d_x_gx_2 = output.data.mean()

        g_optimizer.step()

        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d](%02d/%02d) => '
                  % (epoch + 1, num_epochs,
                     i + 1, len(train_dataset) // batch_size), end=' ')
            print('Loss_D: {:.4f} Loss_G: {:.4f}'
                  ' D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
                      d_loss.data[0], g_loss.data[0], d_x_y, d_x_gx, d_x_gx_2))

    torchvision.utils.save_image(
        fake_b.data, './output/fake_samples_epoch%d.png' % (epoch + 1))

torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')
torch.save(generator, './generator.pth')
